import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from file_operations.File_operations_Azure import File_Operations
import config as cfg

class Send_mail:
    def __init__(self, maillogcollection, logger, subject):
        self.sender_address = cfg.sender_address
        self.sender_pass = cfg.sender_pass
        self.receiver_address = cfg.receiver_address
        self.subject = subject
        self.db_object = maillogcollection
        self.log = logger
        self.fileoperation = File_Operations(self.log)
        

    def send_mail_content(self, data):
        if self.subject=='Training Status':
            mail_content = '''Hello,
            Training Completed Successfully!!!
            Below are the files which were ignored during training
            '''
        else:
            mail_content = '''Hello,
                        Prediction Completed Successfully!!!
                        Below are the files which were ignored during prediction
                        '''
        mail_content = mail_content + '\n'
        #attach files
        files = self.fileoperation.getallFiles(data)

        for index, file in enumerate(files,1):
            mail_content = mail_content + str(index)+')'+' '+ file + '\n'

        #Setup the MIME
        msg = MIMEMultipart()
        msg["From"] = self.sender_address
        msg["To"] = self.receiver_address
        msg["Subject"] = self.subject
        msg.preamble = self.subject
        msg.attach(MIMEText(mail_content, 'plain'))

        # for file in files:
        #     fileToSend = self.fileoperation.downloadfiles(data, file)
        #     ctype, encoding = mimetypes.guess_type(fileToSend)
        #     if ctype is None or encoding is not None:
        #         ctype = "application/octet-stream"
        #
        #     maintype, subtype = ctype.split("/", 1)
        #
        #     if maintype == "text":
        #         fp = open(fileToSend)
        #         # Note: we should handle calculating the charset
        #         attachment = MIMEText(fp.read(), _subtype=subtype)
        #         fp.close()
        #     elif maintype == "image":
        #         fp = open(fileToSend, "rb")
        #         attachment = MIMEImage(fp.read(), _subtype=subtype)
        #         fp.close()
        #     elif maintype == "audio":
        #         fp = open(fileToSend, "rb")
        #         attachment = MIMEAudio(fp.read(), _subtype=subtype)
        #         fp.close()
        #     else:
        #         fp = open(fileToSend, "rb")
        #         attachment = MIMEBase(maintype, subtype)
        #         attachment.set_payload(fp.read())
        #         fp.close()
        #         encoders.encode_base64(attachment)
        #     attachment.add_header("Content-Disposition", "attachment", filename=fileToSend)
        #     msg.attach(attachment)

        server = smtplib.SMTP("smtp.gmail.com:587")
        server.starttls()
        server.login(self.sender_address,self.sender_pass)
        server.sendmail(self.sender_address, self.receiver_address, msg.as_string())
        server.quit()
        self.log.db_log(self.db_object, {"Mail success": "Mail Send Succesfully!!!"})