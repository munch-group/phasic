
from functools import wraps, partial
import time
from pathlib import Path
from IPython.display import display, Markdown, HTML
import base64

def hand_off(target_func):
    """Decorator that forwards all parameters to a another function"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return target_func(*args, **kwargs)
        return wrapper
    return decorator

def download_link(notebook_path):
    nb_path = Path(notebook_path)
    b64 = base64.b64encode(nb_path.read_bytes()).decode()
    # display(HTML(f'<a href="data:application/octet-stream;base64,{b64}" download="{nb_path.name}">Download notebook</a>'))
#     display(HTML(f'''<a href="data:application/octet-stream;base64,{b64}" download="{nb_path.name}" title="Download notebook">
#   <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
#     <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
#     <polyline points="7 10 12 15 17 10"/>
#     <line x1="12" y1="15" x2="12" y2="3"/>
#   </svg>
# </a>'''))
#     display(HTML(f'''<a href="data:application/octet-stream;base64,{b64}" download="{nb_path.name}" title="Download notebook" style="text-decoration:none;">
#   Download notebook
#   <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;">
#     <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
#     <polyline points="7 10 12 15 17 10"/>
#     <line x1="12" y1="15" x2="12" y2="3"/>
#   </svg>
# </a>'''))
    display(HTML(f'''<div style="text-align:right;">
  <a href="data:application/octet-stream;base64,{b64}" download="{nb_path.name}" title="Download notebook" style="text-decoration:none;">
    Download notebook
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="7 10 12 15 17 10"/>
      <line x1="12" y1="15" x2="12" y2="3"/>
    </svg>
  </a>
</div>'''))