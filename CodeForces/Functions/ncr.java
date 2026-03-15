import java.util.Scanner;

public class Main{
    public static long factorial(int n){
        long fact = 1;
        for(int i = 1; i <= n; i++){
            fact *= i;
        }
        return fact;
    }
    
    public static long coeff(int n, int r){
        return factorial(n) / (factorial(r) * factorial(n-r));
    }
    
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        
        int n = sc.nextInt();
        int r = sc.nextInt();
        
        System.out.print(coeff(n, r));
    }
}
