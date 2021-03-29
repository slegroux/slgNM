#!/usr/bin/env bash
ps -aux |grep speech_to_text.py|awk '{ print $2 }'|xargs -I{} kill -9 {}
