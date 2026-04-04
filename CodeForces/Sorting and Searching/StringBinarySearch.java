import java.util.*;

public class Main{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        
        String[] arr = new String[n];
        
        for(int i = 0; i < n; i++){
            arr[i] = sc.nextLine();
        }
        
        String x = sc.nextLine();
        
        int left = 0, right = n-1;
        boolean found = false;
        
        while(left <= right){
            int mid = left + (right - left)/2;
            
            int cmp = arr[mid].compareTo(x);
            
            if(cmp == 0){
                found = true;
                break;
            }
            else if(cmp < 0){
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
