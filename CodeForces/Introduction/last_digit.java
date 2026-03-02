import java.util.Scanner;
public class Main{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        int a = sc.nextInt();
        int b = sc.nextInt();
        
        int c = a % 10;
        int d = b % 10;
        
        System.out.println(c+d);
        
    }
}
