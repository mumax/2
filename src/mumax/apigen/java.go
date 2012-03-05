//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.


package apigen

import (
	"io"
	"reflect"
	"fmt"
)

type Java struct{}

func (p *Java) Filename() string {
	return "Mumax2.java"
}

func (j *Java) Comment() string {
	return "//"
}

func (j *Java) WriteHeader(out io.Writer) {
	fmt.Fprintln(out, `
import java.io.*;

public class Mumax2{
	private static BufferedReader infifo;
	private static PrintStream outfifo;
	private static String outputDir;
	private static boolean initialized;

	private static void init() throws IOException{
		// get the output directory from the environment
		outputDir = System.getenv("MUMAX2_OUTPUTDIR");
		// signal our intent to open the fifos
		new File(outputDir, "handshake").createNewFile();
		// the order in which the fifos are opened matters
		infifo = new BufferedReader(new FileReader(new File(outputDir, "out.fifo"))); // mumax's out is our in
		outfifo = new PrintStream(new FileOutputStream(new File(outputDir, "in.fifo"))); // mumax's in is our out
		initialized = true;
	}

	private static String call(String command, String[] args){
		try{
			if(!initialized){
				init();
			}
			outfifo.print(command);	
			for(int i=0; i<args.length; i++){
				outfifo.print(" ");
				outfifo.print(args[i]);
			}
			outfifo.println();
			return infifo.readLine();
		}catch(IOException e){
			System.err.println(e);
			System.exit(-1);
		}
		System.exit(-2); // unreachable
		return "bug";
	}
`)
}

func (j *Java) WriteFooter(out io.Writer) {
	fmt.Fprint(out, `}`)
}

func (j *Java) WriteFunc(out io.Writer, funcName string, argTypes []reflect.Type, returnType reflect.Type) {
	fmt.Fprintln(out)

	ret := ""
	if returnType != nil {
		ret = returnType.String()
	}

	fmt.Fprintf(out, `
	public static %s %s(`,
		java_type[ret], funcName)

	args := ""
	for i := range argTypes {
		if i != 0 {
			args += ", "
		}
		args += java_type[argTypes[i].String()] + " "
		args += "arg" + fmt.Sprint(i+1)
	}
	fmt.Fprintln(out, args, "){")

	fmt.Fprintf(out, `		String returned = call("%s", new String[]{`, funcName)

	for i := range argTypes {
		if i != 0 {
			fmt.Fprintf(out, ", ")
		}
		fmt.Fprintf(out, `"" + arg%v`, i+1)
	}
	fmt.Fprintln(out, "});")
	if returnType != nil {
		fmt.Fprintf(out, `		return %s(returned);`, java_parse[ret])
		fmt.Fprintln(out)
	}
	fmt.Fprintln(out, `	}`)
}

var (
	// functions for parsing go types from string
	java_parse map[string]string = map[string]string{"int": "Integer.parseInt",
		"float32": "Float.parseFloat",
		"float64": "Double.parseDouble",
		"bool":    "Boolean.parseBoolean"}
	// maps go types onto java types
	java_type map[string]string = map[string]string{"int": "int",
		"float32": "float",
		"float64": "double",
		"string":  "String",
		"bool":    "boolean",
		"":        "void"}
)
