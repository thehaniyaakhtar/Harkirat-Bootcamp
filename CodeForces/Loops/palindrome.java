import java.util.Scanner;

public class Main{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        int a = sc.nextInt();
        int original = a;
        int rev = 0;
        
        while(a > 0){
            int digit = a % 10;
            rev = rev * 10 + digit;
            a = a / 10;
        }
        
        if(original == rev){
            System.out.print("YES");
        }
        else{
            System.out.print("NO");
        }
        
    }
}
