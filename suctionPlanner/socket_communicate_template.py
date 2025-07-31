import socket
class ClientCommunicator:
    def __init__(self):
        self.client_socket = None
        self.client_connected = False
        self.SERVER_IP = "127.0.0.1"
        self.SERVER_PORT = 6666

    def connect_to_server(self):
        """连接到服务端"""
        try:
            if self.client_socket:
                self.client_socket.close()
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.SERVER_IP,self.SERVER_PORT))
            self.client_connected = True
            print(f"已连接到服务端:{self.SERVER_IP}:{self.SERVER_PORT}")
            return True
        except Exception as e:
            print(f"连接服务端失败: {e}")
            self.client_connected = False
            return False

    def disconnect_from_server(self):
        """断开与服务端的连接"""
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
            self.client_socket = None
            self.client_connected = False
            print("已断开与服务连接")

    def wait_for_request_from_server(self):
        """等待服务端发送请求"""
        if not self.client_socket:
            print("服务端未连接")
            return False       
        # 发送计算请求
        try:
            response = self.client_socket.recv(1024).decode('utf-8')
            if response.strip() != "start_calc":
                print(f"接收请求计算失败: {response}")
                return False
            return True
        except Exception as e:
            print(f"等待服务端请求失败: {str(e)}")
            return False

    def reply_depallet_calculation(self, box_size, start_coord, start_euler_z, position_offset):
        """向服务端发送箱子尺寸和拆垛位姿并接收确认数据"""
        try:
            box_size_str = ','.join(map(str, box_size)) + "\r\n"
            # print(f"发送箱子尺寸: {box_size_str.strip()}")
            self.client_socket.sendall(box_size_str.encode('utf-8'))
            response = self.client_socket.recv(1024).decode('utf-8').strip()
            if response.strip() != box_size_str.strip():
                print(f"box_size_str确认失败: {response}")
                return False
            
            start_coord_str = ','.join(map(str, start_coord)) + "\r\n"
            self.client_socket.sendall(start_coord_str.encode('utf-8'))
            response = self.client_socket.recv(1024).decode('utf-8').strip()
            if response.strip() != start_coord_str.strip():
                print(f"start_coord_str确认失败: {response}")
                return False
            
            start_euler_str = ','.join(map(str, [start_euler_z,180,0])) + "\r\n"
            self.client_socket.sendall(start_euler_str.encode('utf-8'))
            response = self.client_socket.recv(1024).decode('utf-8').strip()
            if response.strip() != start_euler_str.strip():
                print(f"start_euler_str确认失败: {response}")
                return False

            position_offset_str = ','.join(map(str, position_offset)) + "\r\n"
            self.client_socket.sendall(position_offset_str.encode('utf-8'))
            response = self.client_socket.recv(1024).decode('utf-8').strip()
            if response.strip() != position_offset_str.strip():
                print(f"position_offset_str确认失败: {response}")
                return False

            print(f"拆垛数据确认成功")
            return True

        except Exception as e:
            print(f"发送数据错误: {str(e)}")
            return False

if __name__ == "__main__":
    client = ClientCommunicator()
    client.connect_to_server()
    client.wait_for_request_from_server()
    client.reply_depallet_calculation([100,100,100],[100,100,100],0,[100,100,100])


            