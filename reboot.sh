#!/usr/bin/expect -f
proc cj_reboot {user ip pass} {log_user 0;
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
# set ip1 "192.168.189.10"
# set user1 "amax4" 
# set pass1 "Rc4ml0624@9$\n"
set ip1 "192.168.189.7"
set user1 "amax1" 
set pass1 "Rc4ml0624@3$\n"

set ip2 "192.168.189.8"
set user2 "amax2" 
set pass2 "Rc4ml0624@5$\n"

set ip3 "192.168.189.9"
set user3 "amax3" 
set pass3 "Rc4ml0624@7$\n"

set ip4 "192.168.189.10"
set user4 "amax4" 
set pass4 "Rc4ml0624@9$\n"

set ip5 "192.168.189.11"
set user5 "amax5" 
set pass5 "Rc4ml0624@11$\n"

set ip6 "192.168.189.12"
set user6 "amax6" 
set pass6 "Rc4ml0624@13$\n"

set ip7 "192.168.189.13"
set user7 "amax7" 
set pass7 "Rc4ml0624@15$\n"

set ip8 "192.168.189.14"
set user8 "amax8" 
set pass8 "Rc4ml0624@17$\n"

set ip9 "192.168.189.25"
set user9 "root" 
set pass9 "Rc4ml0624@wcx\n"

cj_reboot $user4 $ip4 $pass4
sleep 10
cj_ping $ip4
