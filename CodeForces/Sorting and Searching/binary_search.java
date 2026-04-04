import java.util.*;

public class Main{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        
        int n = sc.nextInt();
        int[] arr = new int[n];
        
        for(int i = 0; i < n; i++){
            arr[i] = sc.nextInt();
        }
        
        int x = sc.nextInt();
        
        int left = 0, right = n-1;
        boolean found = false;
        
        while(left <= right){
            int mid = left + (right-left)/2;
            
            if(arr[mid] == x){
                found = true;
                break;
            }
            else if(arr[mid] < x){
                left = mid + 1;
            }
            else{
                right = mid - 1;
            }
        }
        
        if(found){
            System.out.println("YES");
        }
        else{
            System.out.println("NO");
        }
    }
}
