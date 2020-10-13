import os
import pickle

import numpy as np

from recognizers_date_time import DateTimeRecognizer, Culture

import jieba

jieba.initialize()


class DateRecognizer:
    def __init__(self):
 
        self.date_recognizer = None
        self.date_recognizer_model = None

        self.init_date_recognizer()

    def init_date_recognizer(self):
        self.date_recognizer = DateTimeRecognizer()
        self.date_recognizer_model = self.date_recognizer.get_datetime_model(Culture.Chinese)

    def parse_date(self, text):
        result = self.date_recognizer_model.parse(text)
        dates = []
        for r in result:
            print(r.text, r.type_name, r.resolution)
            resolution = r.resolution
            if resolution:
                if resolution['values'][0]['type'] == "daterange":
                    if not resolution['values'][0]['timex'] or resolution['values'][0]['value'] == 'not resolved':
                        continue
                    dates.append({"type": resolution['values'][0]['type'],
                                  "start": resolution['values'][0]['start'],
                                  "end": resolution['values'][0]['end'], "text": r.text})
                elif resolution['values'][0]['type'] == "date":
                    dates.append({"type": resolution['values'][0]['type'],
                                  "entity": resolution['values'][0]['value'], "text": r.text})
                elif resolution['values'][0]['type'] == "datetime":
                    dates.append({"type": resolution['values'][0]['type'],
                                  "entity": resolution['values'][0]['value'], "text": r.text})
                elif resolution['values'][0]['type'] == "duration":
                    dates.append({"type": resolution['values'][0]['type'],
                                  "entity": resolution['values'][0]['timex'].replace('P', '').replace('T', ''), "text": r.text})
        return dates


if __name__ == '__main__':
    recognizer = DateRecognizer()
    print(recognizer.parse_date("预定明天的酒店"))
    print(recognizer.parse_date("下周去北京"))
