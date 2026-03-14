import java.util.Scanner;

public class Main{
    public static void factors(int n){
        for(int i = n; i > 0; i--){
            if(n % i == 0){
                System.out.print(i+" ");
            }
        }
    }
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        factors(n);
    }
}
