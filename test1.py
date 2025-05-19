import requests
import json

def send_wechat_message(content, webhook_url):
    """
    发送文本消息到企业微信机器人群聊
    content: 消息字符串
    webhook_url: 企业微信机器人Webhook地址
    """
    headers = {'Content-Type': 'application/json'}
    data = {
        "msgtype": "text",
        "text": {
            "content": content
        }
    }
    response = requests.post(webhook_url, data=json.dumps(data), headers=headers)
    if response.status_code == 200:
        resp_json = response.json()
        if resp_json.get("errcode") == 0:
            print("企业微信消息发送成功")
        else:
            print("企业微信消息发送失败，错误信息:", resp_json)
    else:
        print("请求失败，状态码:", response.status_code)

if __name__ == "__main__":
    # 替换成你的企业微信机器人Webhook地址
    webhook = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=af258228-be3e-40bc-9321-0d4d1d0cb9d9"
    send_wechat_message("Python程序运行完成！", webhook)
