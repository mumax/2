package main

import (
	"flag"
	"fmt"
)


var (
	nodes []*Node
	queue []string

	
	//done chan(*Task)
)

func init(){

}

func main() {
	flag.Parse()

	setup()

	PrintInfo()
}


func PrintInfo(){
	PrintNodes()
}


func PrintNodes(){
	for _, n := range nodes{
		fmt.Println(n)
	}
}

func setup(){
	localhost := NewNode("localhost")
	localhost.AddGPU(NewGPU(512))
	localhost.AddGPU(NewGPU(512))
	AddNode(localhost)
}

func AddNode(n *Node) {
	nodes = append(nodes, n)
}
