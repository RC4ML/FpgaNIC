#!/usr/bin/expect -f
set user "amax3"
set ip "192.168.189.9"

spawn ssh "$user@$ip" ; log_user 0;
expect "]*"
send "sudo rm -rf /home/$user/xdma\n"
expect "]*"
send "exit\n"
expect eof

spawn rsync -av --exclude '.git'  /home/amax/cj/xdma/ "$user@$ip:/home/$user/xdma/"; log_user 0;
expect "]*"
sleep 3
spawn ssh "$user@$ip"
expect "]*"
send "cd /home/$user/xdma/sw/src/\n" ;log_user 1;
expect "]*"
send "mkdir build\n"
expect "]*"
send "cd build\n"
expect "]*"
send "cmake ..\n"
expect "]*"
send "make\n"
expect eof
expect eof
send "sudo ./dma-example -t client -n 3\n"
expect "]*"
interact