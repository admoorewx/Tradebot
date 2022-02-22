import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email(content):
    message = MIMEMultipart()
    message["From"] = ""
    message["To"] = ""
    message["Subject"] = "Daily Tradebot Report"
    message.attach(MIMEText(content, "plain"))
    message = message.as_string()
    port = ###
    password = ""
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("", port, context=context) as server:
        server.login("",password)
        server.sendmail("","",message)
