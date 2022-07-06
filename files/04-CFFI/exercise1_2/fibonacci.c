#include <stdio.h>
#include <stdlib.h>

void fibo(int *a, int *b);

int main(){

    int n,i;
    int n1=1,n2=0;
    
    printf("Enter a positive integer\n");
    scanf("%d",&n);
    
    if (n < 1) {
        printf("The number is not positive\n");
    }
    
    printf("The fibonacci sequence is : \n");
    printf("%d, %d, ",n2,n1);
    
    for (i=2;i<n;i++) {
        fibo(&n1, &n2);
        printf("%d, ", n1);
    }
    
    fibo(&n1, &n2);
    printf("%d\n", n1);
    
}

void fibo(int *a, int *b)
{
  int next;
  next = *a + *b;
  *a = *b;
  *b = next;
}
