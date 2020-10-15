#!/usr/bin/expect -f
spawn ssh amax4@192.168.189.10 ; log_user 0;
expect "]*"
send "sudo rm -rf /home/amax4/xdma\n"
expect "]*"
send "Rc4ml0624@9$\n"
expect "]*"
send "exit\n"
expect eof

spawn scp -r /home/amax/cj/xdma amax4@192.168.189.10:/home/amax4; log_user 0;
expect "]*"
spawn ssh amax4@192.168.189.10 
expect "]*"
send "cd /home/amax4/xdma/sw/src/\n" ;log_user 1;
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
send "sudo ./dma-example\n"
expect "]*"
send "Rc4ml0624@9$\n"
interact
