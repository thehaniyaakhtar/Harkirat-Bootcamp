import java.util.Scanner;

public class Main{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        int a = sc.nextInt();
        int prod = 1;
        
        for(int i = 1; i <= a; i++){
            prod = prod*i;
        }
        System.out.println(prod);
    }
}
