/* Optional tab-scoped access memory. Never persist in localStorage or URL. */
(() => {
  const token = document.getElementById('token');
  const remember = document.getElementById('rememberToken');
  const clear = document.getElementById('clearToken');
  if (!token || !remember || !clear) return;
  const key = 'research-copilot.tab-token.v1';
  try {
    const saved = sessionStorage.getItem(key);
    if (saved) { token.value = saved; remember.checked = true; }
  } catch (_) { remember.disabled = true; }
  function sync() {
    try {
      if (remember.checked && token.value.trim()) sessionStorage.setItem(key, token.value.trim());
      else sessionStorage.removeItem(key);
    } catch (_) { remember.checked = false; remember.disabled = true; }
  }
  token.addEventListener('input', sync);
  remember.addEventListener('change', sync);
  clear.addEventListener('click', () => { token.value = ''; remember.checked = false; sync(); });
})();
