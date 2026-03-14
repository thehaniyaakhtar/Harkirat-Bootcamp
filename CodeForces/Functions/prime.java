import java.util.Scanner;

public class Main{
    public static void prime(int n){
        boolean prime = true;
        
        if(n<=1){
            prime = false;
        }
        
        for(int i = 2; i <= n-1; i++){
            if(n%i == 0){
                prime = false;
                break;
            }
        }
        
        if(prime){
            System.out.print("Prime");
        }
        else{
            System.out.print("Not Prime");
        }
    }
    
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        prime(n);
    }
}
