import java.util.Scanner;

public class Main{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        long a = sc.nextLong();
        long rev = 0;
        
        while(a>0){
            long digit = a % 10;
            rev = rev * 10 + digit;
            a = a /10;
        }
        System.out.print(rev);
    }
}
