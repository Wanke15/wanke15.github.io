```python
import multiprocessing
from multiprocessing import Manager

from functools import partial

def process_info(info, res_dict):
  name = info.get('name')
  upper_name = name.upper()
  res_dict[name] = upper_name
  
if __name__ == '__main__':
  manager = Manager()
  share_dict = manager.dict()
  names = ['jeff', 'tina']

  pool = multiprocessing.Pool(4)
  pool.map(partial(process_info, res_dict=share_dict), names)
  pool.close()
  pool.join()

  print(share_dict)
```
