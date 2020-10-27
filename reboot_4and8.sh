#!/usr/bin/expect -f
proc cj_reboot {user ip pass} {log_user 0;
	spawn ssh "$user@$ip" ;
	expect "]*"
	send "sudo reboot\n"; log_user 1;
	expect "]*"
	send $pass
	expect "]*"
}

proc cj_ping {ip} {log_user 0;
	spawn ping -c 2 -i 4 -W 1 $ip
	expect  {
        " 0%" {puts $ip:连接成功; }
		" 50%" {puts $ip:未连接; cj_ping $ip}
        " 100%" {puts $ip:未连接; cj_ping $ip}
        }
}
# spawn ping -c 2 -i 3 -W 1 192.168.189.10
set ip1 "192.168.189.10"
set user1 "amax4" 
set pass1 "Rc4ml0624@9$\n"

set ip2 "192.168.189.14"
set user2 "amax8" 
set pass2 "Rc4ml0624@17$\n"

cj_reboot $user1 $ip1 $pass1
cj_reboot $user2 $ip2 $pass2

cj_ping $ip1
cj_ping $ip2
