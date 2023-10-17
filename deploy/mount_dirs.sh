#!/bin/bash
# Mount image folders (read-only)
sshfs ecdysis@ecdysis01.local:/pool1/srv/bugbox/ /pool1/srv/bugbox/ -o ro
# Mount model shared folder for deploy (write access)
sshfs ecdysis@ecdysis01.local:/pool1/model-store/ /pool1/model-store/
sshfs ecdysis@ecdysis01.local:/pool1/smb/ /pool1/smb/
