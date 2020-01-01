import zmail
from typing import List


class MyEmailServer:

    @staticmethod
    def send_mail(subject: str, content_text: str, to_addr: List[str] or str) -> bool:
        # 配置发送方的邮箱和密码
        server = zmail.server('buddaa@163.com', 'xxxxxx')
        # 邮件的主题和内容
        mail_content = {'subject': subject, 'content_text': content_text}
        
        return server.send_mail(to_addr, mail_content)

if __name__ == '__main__':
    result = MyEmailServer.send_mail("般若波罗蜜多心经",
                            "观自在菩萨，行深般若波罗蜜多时，照见......",
                            ['buddaa@foxmail.com', 'buddaa@126.com'])

    print(result)
