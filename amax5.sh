#!/usr/bin/expect -f
set user "amax5"
set ip "192.168.189.11"
set pass "Rc4ml0624@11$\n"

spawn ssh "$user@$ip" ; log_user 0;
expect "]*"
send "sudo rm -rf /home/$user/xdma\n"
expect "]*"
send $pass
expect "]*"
send "exit\n"
expect eof

spawn scp -r /home/amax/cj/xdma "$user@$ip:/home/$user"; log_user 0;
expect "]*"
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
send "sudo ./dma-example -t server -n 4\n"
expect "]*"
send $pass
interact