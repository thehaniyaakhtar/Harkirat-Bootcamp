import java.util.Scanner;

public class Main{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        
        int a = sc.nextInt();
        int b = sc.nextInt();
        
        if(a > b){
            System.out.println("Min = " + b + "\nMax = " + a);
        }
        else{
            System.out.println("Min = " + a + "\nMax = " + b);
        }
    }
}
