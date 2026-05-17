# Local development utilities. Project itself is uv-managed; this Makefile
# is just for one-shot ops commands that don't fit the `uv run` pattern.

.PHONY: monitor monitor-stop monitor-restart monitor-logs \
        monitor-docker monitor-docker-stop monitor-docker-logs

# Default path: Netdata installed natively via the kickstart script
# (see README → "Live system monitor"). After install it runs as a
# systemd service and auto-starts on boot, so most of these targets are
# just convenient wrappers around `systemctl` / `journalctl`.

# Status + dashboard URL.
monitor:
	@echo "Dashboard: http://localhost:19999"
	@echo "Tailscale: http://$$(hostname):19999 from any device on your Tailnet"
	@echo
	@systemctl is-active --quiet netdata 2>/dev/null && echo "netdata.service is active" \
	  || echo "netdata.service is not active (start with: sudo systemctl start netdata)"

monitor-stop:
	sudo systemctl stop netdata

monitor-restart:
	sudo systemctl restart netdata

monitor-logs:
	journalctl -u netdata -f

# Alternative: containerized Netdata. Use if you'd rather not install
# the agent system-wide. `make monitor-docker` brings it up;
# `monitor-docker-stop` removes it (the named volumes survive).
monitor-docker:
	docker run -d --name=netdata --restart=unless-stopped \
	  -p 19999:19999 --pid=host \
	  -v netdataconfig:/etc/netdata \
	  -v netdatalib:/var/lib/netdata \
	  -v netdatacache:/var/cache/netdata \
	  -v /etc/passwd:/host/etc/passwd:ro \
	  -v /etc/group:/host/etc/group:ro \
	  -v /proc:/host/proc:ro \
	  -v /sys:/host/sys:ro \
	  -v /etc/os-release:/host/etc/os-release:ro \
	  --cap-add SYS_PTRACE --cap-add SYS_ADMIN \
	  --security-opt apparmor=unconfined \
	  --gpus all \
	  netdata/netdata
	@echo
	@echo "Netdata is starting. Dashboard at http://localhost:19999"

monitor-docker-stop:
	docker rm -f netdata

monitor-docker-logs:
	docker logs -f netdata
