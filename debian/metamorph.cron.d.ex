#
# Regular cron jobs for the metamorph package
#
0 4	* * *	root	[ -x /usr/bin/metamorph_maintenance ] && /usr/bin/metamorph_maintenance
