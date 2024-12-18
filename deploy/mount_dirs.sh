#!/bin/bash
# Mount image folders (read-only)
sshfs ecdysis@ecdysis01.local:/pool1/srv/bugbox3/ /pool1/srv/bugbox3/ -o ro

# or was this done?
# sudo sshfs ecdysis@ecdysis01.local:/pool1/srv/bugbox3/bugbox3/media/ /pool1/srv/bugbox3/bugbox3/media/ -o ro

# Mount model shared folder for deploy (write access)
sshfs ecdysis@ecdysis01.local:/pool1/srv/bugbox3/local_files /pool1/srv/bugbox3/local_files
sshfs ecdysis@ecdysis01.local:/pool1/model-store-2/ /pool1/model-store-2/
sshfs ecdysis@ecdysis01.local:/pool1/smb/ /pool1/smb/
