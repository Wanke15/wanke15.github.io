import java.util.*;

public class DiversityReorder {

    // 定义一个Item类，包含id和属性
    static class Item {
        int id;
        String attribute;

        public Item(int id, String attribute) {
            this.id = id;
            this.attribute = attribute;
        }

        @Override
        public String toString() {
            return "Item{id=" + id + ", attribute='" + attribute + "'}";
        }
    }

    public static void reorderItems(List<Item> items, int windowSize, int maxSameAttributeCount) {
        Map<String, Integer> attributeCountMap = new HashMap<>();
        Queue<Item> window = new LinkedList<>();

        for (int i = 0; i < items.size(); i++) {
            Item currentItem = items.get(i);

            // 更新滑动窗口
            if (window.size() == windowSize) {
                Item removedItem = window.poll();
                attributeCountMap.put(removedItem.attribute, attributeCountMap.get(removedItem.attribute) - 1);
            }

            // 检查当前item是否符合要求
            if (attributeCountMap.getOrDefault(currentItem.attribute, 0) < maxSameAttributeCount) {
                window.offer(currentItem);
                attributeCountMap.put(currentItem.attribute, attributeCountMap.getOrDefault(currentItem.attribute, 0) + 1);
            } else {
                // 从剩下的物品中选择一个符合要求的物品
                boolean found = false;
                for (int j = i + 1; j < items.size(); j++) {
                    Item remainingItem = items.get(j);
                    if (attributeCountMap.getOrDefault(remainingItem.attribute, 0) < maxSameAttributeCount) {
                        // 交换位置
                        Collections.swap(items, i, j);
                        window.offer(items.get(i));
                        attributeCountMap.put(items.get(i).attribute, attributeCountMap.getOrDefault(items.get(i).attribute, 0) + 1);
                        found = true;
                        break;
                    }
                }
                // 如果没有找到符合要求的物品，则不处理当前物品
                if (!found) {
                    window.offer(currentItem);
                    attributeCountMap.put(currentItem.attribute, attributeCountMap.getOrDefault(currentItem.attribute, 0) + 1);
                }
            }
        }
    }

    public static void main(String[] args) {
        List<Item> items = Arrays.asList(
                new Item(1, "A"),
                new Item(2, "A"),
                new Item(3, "B"),
                new Item(4, "A"),
                new Item(5, "B"),
                new Item(6, "C"),
                new Item(7, "A"),
                new Item(8, "C")
        );

        int windowSize = 3;
        int maxSameAttributeCount = 1;

        reorderItems(items, windowSize, maxSameAttributeCount);

        for (Item item : items) {
            System.out.println(item);
        }
    }
}