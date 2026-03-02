import java.util.Scanner;

public class Main{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        
        int a = sc.nextInt();
        int b = sc.nextInt();
        int c = sc.nextInt();
        
        int min = Math.min(a, Math.min(c, b));
        int max = Math.max(a, Math.max(b, c));
        
        System.out.println("Min = " + min);
        System.out.println("Max = " + max);
    }
}
