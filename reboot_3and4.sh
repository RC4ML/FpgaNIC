#!/usr/bin/expect -f
proc cj_reboot {user ip} {log_user 0;
	spawn ssh "$user@$ip" ;
	expect "]*"
	send "sudo reboot\n"; log_user 1;
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


set ip3 "192.168.189.9"
set user3 "amax3" 

set ip4 "192.168.189.10"
set user4 "amax4" 
cj_reboot $user3 $ip3
cj_reboot $user4 $ip4

sleep 10
cj_ping $ip3
cj_ping $ip4