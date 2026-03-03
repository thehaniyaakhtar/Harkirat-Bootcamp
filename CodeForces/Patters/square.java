import java.util.Scanner;

public class Main{
    public static void main(String[] args){
        
        Scanner sc = new Scanner(System.in);
        
        int a = sc.nextInt();
        int b = a;
        
        for(int i = 0; i < a; i++){
            for(int j = 0; j < b; j++){
                System.out.print("*");
            }
            System.out.println();
        }
            
    }
    
}
