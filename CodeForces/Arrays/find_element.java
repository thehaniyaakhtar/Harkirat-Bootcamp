import java.util.Scanner;

public class main{
    public static void main(String[] args){
        
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int b = sc.nextInt();
        int arr[] = new int[n];
        
        boolean found = false;
        
        for(int i = 0; i < n; i++){
            arr[i] = sc.nextInt();
            
            if(arr[i] == b){
                found = true;
            }
        }
        
        if(found){
            System.out.print("YES");
        }
        else{
            System.out.print("NO");
        }
    }
}
