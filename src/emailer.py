import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email(content):
    message = MIMEMultipart()
    message["From"] = "gcetradebot@gmail.com"
    message["To"] = "admoorewx@gmail.com"
    message["Subject"] = "Daily Tradebot Report"
    message.attach(MIMEText(content, "plain"))
    message = message.as_string()
    port = 465
    password = "Tradebot1!"
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login("gcetradebot@gmail.com",password)
        server.sendmail("gcetradebot@gmail.com","admoorewx@gmail.com",message)