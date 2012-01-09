//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Implementation of "adduser" command

import (
	. "mumax/common"
)

func init() {
	api["adduser"] = adduser
}

const DEFAULT_SHARE = 1 // default share of one device per user

// adds a user
func adduser(user *User, args []string) string {
	if len(args) < 2 {
		return "Usage: adduser <username> <group> [share]"
	}
	username := args[0]
	group := args[1]
	share := DEFAULT_SHARE
	if len(args) == 3 {
		share = Atoi(args[2])
	}
	AddUser(username, group, share)
	return "added user " + GetUser(username).LongString()
}
