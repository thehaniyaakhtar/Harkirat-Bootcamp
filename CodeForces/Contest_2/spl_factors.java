import java.util.Scanner;

public class Main{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        boolean found = true;
        
        for(int i = 1; i <= n; i++){
            if(n%i==0){
                if(i%10 == 2 || i % 10 == 7){
                    System.out.print(i + " ");
                    found = true;
                }
            }
        }
        if(!found) System.out.print("-1");
    }
}
