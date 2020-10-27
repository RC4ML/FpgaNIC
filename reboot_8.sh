#!/usr/bin/expect -f
spawn ssh amax8@192.168.189.14 ; log_user 0;
expect "]*"
send "sudo reboot\n"; log_user 1;
expect "]*"
send "Rc4ml0624@17$\n"
expect "]*"