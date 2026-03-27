import java.util.Scanner;

public class Main{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        
        long n = sc.nextLong();
        long m = sc.nextLong();
        
        if(n==m || (n <= 1 && m <= 1)){
            System.out.print("Yes");
        }
        else{
            System.out.print("No");
        }
    }
}
