import java.util.Scanner;

public class Main{
    public static void main(String[] args){
        
        Scanner sc = new Scanner(System.in);
        long a = sc.nextLong();
        
        if(a==0){
            System.out.print(0);
        }
        
        while(a!=0){
            long digit = a%10;
            System.out.print(digit);
            a = a / 10;
        }
        
    }
}
