import pysftp

class Sftp_Helper():
    """SFTP Handler"""
    def __init__(self, host = 'emimdgxa100gpu1.ccds.io', port=22):
        
        self.host = host 
        self.port = 22 
        self.username = 'sk1064'
        self.password = 'Nada3025!!' 

        self.hostkeys = None

        self.cnopts = pysftp.CnOpts()
        
    def Transfer_data(self, source_path, dest_path):
        if self.cnopts.hostkeys.lookup(self.host) == None:
            print("Hostkey for " + self.host + " doesn't exist")
            hostkeys = self.cnopts.hostkeys 
            self.cnopts.hostkeys = None


        # sftp 접속을 실행
        with pysftp.Connection(
                                self.host,
                                port = self.port,
                                username = self.username,
                                password = self.password,
                                cnopts = self.cnopts) as sftp:
            
            if self.hostkeys != None:
                print("New Host. Caching hostkey for " + self.host)
                self.hostkeys.add(self.host, sftp.remote_server_key.get_name(), sftp.remote_server_key) # 호스트와 호스트키를 추가
                self.hostkeys.save(pysftp.helpers.known_hosts()) # 새로운 호스트 정보 저장

<<<<<<< Updated upstream
            sftp.put(source_path, dest_path)
=======
            sftp.put('./test.png','/home/local/PARTNERS/sk1064/workspace/control/sam_t/test.png')
>>>>>>> Stashed changes
            # 모든 작업이 끝나면 접속 종료
            sftp.close()
        return 
    