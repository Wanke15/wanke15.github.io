```python
import logging

logging.basicConfig(format='[%(levelname)s %(process)d %(thread)d %(processName)s %(threadName)s %(asctime)s %(module)s:%(lineno)d] %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.info("Hello world")
```
