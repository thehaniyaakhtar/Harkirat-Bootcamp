import java.util.Scanner;

public class Main{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        int a = sc.nextInt();
        int b = sc.nextInt();
        long prod = 1;
        
        for(int i = 0; i < b; i++){
            prod = prod * a;;
        }
        System.out.println(prod);
    }
}
