import java.util.Scanner;

public class Main{
    public static void main(String[] args){
        
        Scanner sc = new Scanner(System.in);
        int a = sc.nextInt();
        
        while(a-- > 0){
            int n = sc.nextInt();
            
            for(int i = 1; i <= n; i++){
                if(i % 2 != 0){
                    System.out.print(i + " ");
                }
            }
            
            for(int i = n; i >= 2; i--){
                if(i % 2 == 0){
                    System.out.print(i + " ");
                }
            }
            System.out.println();
        }
    }
}
