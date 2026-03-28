import java.util.Scanner;

public class Main{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        
        int n = sc.nextInt();

        
        for(int i = 1; i <=n n; i++){
            int factor = 0;
            
            for(int j = 1; j <= i; j++){
                if(i%j==0) factor++;
            }
            
            if(factor <= 4){
                System.out.print(i + " ");
            }
        }
    }
}
