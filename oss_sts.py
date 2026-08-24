import os
import json
from dotenv import load_dotenv
from aliyunsdkcore.client import AcsClient
from aliyunsdksts.request.v20150401 import AssumeRoleRequest


"""！！！！！    OSS上传是否有必要，直接传输是否效率更高，因为目前OSS只上传原始文档，并未做任何处理"""
load_dotenv(override=True)
#使用永久密钥向阿里云请求临时令牌
#返回临时 AccessKeyId,AccessKeySecret,SecurityToken，以及bucket信息
def get_oss_token():
    client=AcsClient(
        os.getenv("OSS_ACCESS_KEY_ID"),
        os.getenv("OSS_ACCESS_KEY_SECRET"),
        os.getenv("OSS_REGION")
    )

    request=AssumeRoleRequest.AssumeRoleRequest()
    request.set_RoleArn(os.getenv("OSS_ROLE_ARN"))
    request.set_RoleSessionName("research_copilot_session")
    request.set_DurationSeconds(3600)   #临时有效期为1小时

    response=client.do_action_with_exception(request)
    credentials=json.loads(response)['Credentials']

    return {
        "access_key_id":credentials['AccessKeyId'],
        "access_key_secret":credentials['AccessKeySecret'],
        "security_token":credentials['SecurityToken'],
        "bucket":os.getenv("OSS_BUCKET"),
        "region":os.getenv("OSS_REGION"),
        "endpoint":os.getenv("OSS_ENDPOINT").strip()
    }