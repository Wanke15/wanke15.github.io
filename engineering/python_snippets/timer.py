import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行函数
        end_time = time.time()  # 记录结束时间
        print(f"函数 {func.__name__} 耗时: {end_time - start_time:.4f} 秒")
        return result
    return wrapper
