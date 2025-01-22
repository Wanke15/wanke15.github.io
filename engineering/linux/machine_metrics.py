from datetime import datetime, timedelta

import requests
import time


import base64
import hashlib

class Robot:
    def __init__(self):
        self.web_hook = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxx"
        # self.web_hook_group = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxx'
        self.req_json = {}

    def get_cur_time(self):
        cur_day = time.strftime("%Y-%m-%d", time.localtime())
        cur_hour = time.strftime("%H:%M:%S", time.localtime())
        return cur_day, cur_hour

    def send_md_message(self,message, users=['all']):
        """
        发送信息
        :param message:
        :return:
        """
        md_msg_template = {"msgtype": "markdown", "markdown": {"content": ""}}

        cur_day, cur_hour = self.get_cur_time()

        ## 文本信息
        final_msg = message + ",".join(['<@{}>'.format(u) for u in users])
        base_mark = """{} {}\n{}""".format(cur_day, cur_hour, final_msg)

        md_msg_template["markdown"]["content"] = base_mark

        requests.post(self.web_hook, json=md_msg_template)

    def send_auc(self,auc,logloss,label='ctr'):
        auc = "{:.5f}".format(auc)
        logloss = "{:.5f}".format(logloss)
        msg = f'{label}_AUC:{auc},{label}_logloss:{logloss}'
        print(msg)
        self.send_md_message(msg)


    def get_media_id(self,filename):
        id_url = "{}&type=file".format(self.web_hook.replace("send", "upload_media"))
        with open(filename, 'rb') as f:
            content = f.read()
        _data = {filename.split('/')[-1]: content}
        mess = requests.post(id_url, files=_data)
        return mess.json().get("media_id", "")

    def send_file(self,file_save_path):
        media_id = self.get_media_id(file_save_path)
        file_msg = {"msgtype": "file", "file": {"media_id": media_id}}
        requests.post(self.web_hook, json=file_msg)

    def send_pic(self, pic_path):
        """
        企业微信机器人发送图片
        :param pic_path: 图片路径
        :return:
        """
        self.req_json["msgtype"] = "image"
        self.req_json["image"] = {}
        self.req_json["image"]["base64"] = self.path2base64(pic_path)
        self.req_json["image"]["md5"] = self.path2md5(pic_path)
        try:
            resp = requests.post(self.web_hook, json=self.req_json)
            print(f"企业微信机器人发送图片成功")
        except Exception:
            print(f"企业微信机器人发送图片失败")

    def path2base64(self,path):
        """
        文件转换为base64
        :param path: 文件路径
        :return:
        """
        with open(path, "rb") as f:
            byte_data = f.read()
        base64_str = base64.b64encode(byte_data).decode("ascii")    # 二进制转base64
        return base64_str

    def path2md5(self,path):
        """
        文件转换为md5
        :param path: 文件路径
        :return:
        """
        with open(path, "rb") as f:
            byte_data = f.read()
        md5_str = self.md5(byte_data)
        return md5_str

    def md5(self,text):
        """
        md5加密
        :param text:
        :return:
        """
        m = hashlib.md5()
        m.update(text)
        return m.hexdigest()


class GroupRobot(Robot):
    def __init__(self):
        super().__init__()
        self.web_hook = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxx'


# robot = Robot()
robot = GroupRobot()

import subprocess
import psutil

def get_top_process_details():
    # 调用 top 命令，并获取输出
    top_output = subprocess.check_output(['top', '-b', '-n', '1']).decode('utf-8')
    
    # 解析 top 输出，找到 CPU 占用最大的进程
    lines = top_output.split('\n')
    process_lines = lines[7:]  # 跳过前面的标题行
    max_cpu_usage = 0
    max_cpu_pid = None
    
    for line in process_lines:
        if line.strip() == '':
            continue
        parts = line.split()
        if len(parts) < 12:
            continue
        try:
            pid = int(parts[0])
            cpu_usage = float(parts[8])
            if cpu_usage > max_cpu_usage:
                max_cpu_usage = cpu_usage
                max_cpu_pid = pid
        except (ValueError, IndexError):
            continue
    
    if max_cpu_pid is not None:
        # 使用 psutil 获取进程的详细信息
        process = psutil.Process(max_cpu_pid)
        details = {
            'pid': process.pid,
            'username': process.username(),
            'cpu_usage': process.cpu_percent(interval=1.0),
            'memory_usage': process.memory_percent(),
            'command': ' '.join(process.cmdline())
        }
        return details
    else:
        return None

#details = get_top_process_details()
#if details:
#    print(f"CPU 占用最大的进程详情:")
#    print(f"PID: {details['pid']}")
#    print(f"用户名: {details['username']}")
#    print(f"CPU 使用率: {details['cpu_usage']}%")
#    print(f"内存使用率: {details['memory_usage']}%")
#    print(f"命令: {details['command']}")
#else:
#    print("未找到 CPU 占用最大的进程")


import psutil
import time


# 定义阈值和持续时间
THRESHOLD = 90
DURATION = 300

def check_metrics():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    disk_info = psutil.disk_usage('/')

    memory_usage = memory_info.percent
    disk_usage = disk_info.percent

    return cpu_usage, memory_usage, disk_usage

def main():
    cpu_alert = 0
    memory_alert = 0
    disk_alert = 0

    while True:
        cpu_usage, memory_usage, disk_usage = check_metrics()

        if cpu_usage > THRESHOLD:
            cpu_alert += 1
        else:
            cpu_alert = 0

        if memory_usage > THRESHOLD:
            memory_alert += 1
        else:
            memory_alert = 0

        if disk_usage > THRESHOLD:
            disk_alert += 1
        else:
            disk_alert = 0

        msg = ""
        if cpu_alert >= DURATION:
            msg += f"""<font color="red">告警</font>：CPU 使用率超过 {THRESHOLD}% 持续 {DURATION} 秒。当前使用率：{cpu_usage}% \n"""
            print(f"告警：CPU 使用率超过 {THRESHOLD}% 持续 {DURATION} 秒。当前使用率：{cpu_usage}%\n")

            details = get_top_process_details()
            if details and details.get('username') not in ('search'):
                msg += f"CPU 占用最大的进程详情:\n"
                msg += f"PID: {details['pid']}\n"
                msg += f"用户名: {details['username']}\n"
                msg += f"CPU 使用率: {details['cpu_usage']}%\n"
                msg += f"内存使用率: {details['memory_usage']}%\n"
                #msg += f"命令: {details['command']}\n"

            cpu_alert = 0

        if memory_alert >= DURATION:
            msg += f"""<font color="red">告警</font>：内存使用率超过 {THRESHOLD}% 持续 {DURATION} 秒。当前使用率：{memory_usage}% \n"""
            print(f"告警：内存使用率超过 {THRESHOLD}% 持续 {DURATION} 秒。当前使用率：{memory_usage}%")
            memory_alert = 0

        if disk_alert >= DURATION:
            msg += f"""<font color="red">告警</font>：磁盘使用率超过 {THRESHOLD}% 持续 {DURATION} 秒。当前使用率：{disk_usage}% \n"""
            print(f"告警：磁盘使用率超过 {THRESHOLD}% 持续 {DURATION} 秒。当前使用率：{disk_usage}%")
            disk_alert = 0

        if msg:
            msg = "服务器要炸了，赶紧看看吧:\n" + msg
            robot.send_md_message(msg)
        # 每秒检查一次
        time.sleep(1)

if __name__ == "__main__":
    main()
