;Dustin Dodson
;CS441 - System Architecture
;HW0 Part 2
;Due: January 19, 2016

;Requirements Description:
;Use a loop to add up the numbers from 0 to 5, inclusive, and returns this sum.  Run the code, and verify the answer is 15.  Copy the code out as a plain text file named "2.asm".

;Netrun Hardware Description:
;Language: Assembly-NASM
;Mode: Inside a Function
;Function: int foo(void)
;Machine: x86_64 Q6600 x4

push rbx
mov ebx,0
mov eax,0

loop:
add eax, ebx
add ebx, 1
cmp ebx, 6
jne loop

pop rbx
ret