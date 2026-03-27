import java.util.Scanner;
 
public class Main{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int[] arr = new int[n];
        
        for(int i = 0; i < n; i++){
            arr[i] = sc.nextInt();
        }
        
        int left = n / 2 - 1;
        int right = n / 2;
        
        while(left >= 0 && right < n){
            System.out.print(left + " ");
            System.out.print(right + " ");
            left--;
            right++;
        }
    }
}
