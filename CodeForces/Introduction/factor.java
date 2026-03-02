import java.util.Scanner;

public class Main{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        
        int a = sc.nextInt();
        int b = sc.nextInt();
        
        if(b != 0 && a % b == 0){
            System.out.println("Yes");
        }
        else{
            System.out.println("No");
        }
    }
}
