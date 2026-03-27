import java.util.Scanner;

public class Main{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        
        int n = sc.nextInt();

        int zero = 0, one = 0, two = 0;
        
        for(int i = 0; i < n; i++){
            int x = sc.nextInt();
            
            if(x == 0) zero++;
            else if(x==1) one++;
            else two++;
        }
        while(two-- > 0){
            System.out.print("2 ");
        }
        while(one -- > 0){
            System.out.print("1 ");
        }
        while(zero -- > 0){
            System.out.print("0 ");
        }
    }
}
