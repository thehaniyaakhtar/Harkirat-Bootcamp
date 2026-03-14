import java.util.Scanner;

public class Main{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        int a = sc.nextInt();
        count  = 0;
        
        for(int i = 0; i < a; i++){
            int x = sc.nextInt();
            
            if(x != 18 % x == 0 || 45 % x == 0){
                count++
            }
        }
        System.out.print(count);
    }
}
